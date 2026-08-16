import { Pipe, PipeTransform, inject } from '@angular/core';
import { AuthService } from '../services/auth.service';

@Pipe({ name: 'authUrl', standalone: true })
export class AuthUrlPipe implements PipeTransform {
  private auth = inject(AuthService);

  transform(url: string | null | undefined): string | null {
    if (!url) {
      return null;
    }
    const token = this.auth.getToken();
    if (!token) {
      return url;
    }
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}token=${encodeURIComponent(token)}`;
  }
}
