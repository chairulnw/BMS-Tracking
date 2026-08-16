import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Pengaturan } from './pengaturan';

describe('Pengaturan', () => {
  let component: Pengaturan;
  let fixture: ComponentFixture<Pengaturan>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Pengaturan]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Pengaturan);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
