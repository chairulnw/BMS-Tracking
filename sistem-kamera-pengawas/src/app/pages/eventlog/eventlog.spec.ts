import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Eventlog } from './eventlog';

describe('Eventlog', () => {
  let component: Eventlog;
  let fixture: ComponentFixture<Eventlog>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Eventlog]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Eventlog);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
